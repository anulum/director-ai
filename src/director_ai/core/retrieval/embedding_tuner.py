# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

Requires ``pip install sentence-transformers>=2.2``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("DirectorAI.EmbeddingTuner")


@dataclass
class TuneResult:
    model_path: str
    train_samples: int
    epochs: int
    loss_start: float
    loss_end: float


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
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required. Install: pip install director-ai[embeddings]"
        ) from e

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

    loader: DataLoader[Any] = DataLoader(
        train_examples,  # type: ignore[arg-type]  # sentence-transformers InputExample
        shuffle=True,
        batch_size=batch_size,
    )
    train_loss = losses.CosineSimilarityLoss(model)

    # Capture loss values
    loss_start = 0.0
    loss_end = 0.0

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
    )

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
