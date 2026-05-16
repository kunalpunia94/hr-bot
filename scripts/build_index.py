"""Build the FAISS vector index from the cached catalog.

Run once before starting the server (or as part of the deploy build command):
    python -m scripts.build_index
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog_loader import load_catalog  # noqa: E402
from app.config import (  # noqa: E402
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    METADATA_PATH,
    configure_logging,
)

logger = logging.getLogger(__name__)


def assessment_to_text(item: dict) -> str:
    """Compose the searchable text representation of a catalog item."""
    name = item.get("name", "")
    description = item.get("description", "") or ""
    keys = ", ".join(item.get("keys", []) or [])
    job_levels = ", ".join(item.get("job_levels", []) or [])
    languages = ", ".join((item.get("languages") or [])[:8])
    duration = item.get("duration", "") or ""
    remote = item.get("remote", "") or ""
    adaptive = item.get("adaptive", "") or ""
    return (
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"Test categories (keys): {keys}\n"
        f"Job levels: {job_levels}\n"
        f"Languages: {languages}\n"
        f"Duration: {duration} | Remote: {remote} | Adaptive: {adaptive}"
    )


def build() -> None:
    configure_logging()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    logger.info("Loaded %d catalog items", len(catalog))

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [assessment_to_text(item) for item in catalog]
    logger.info("Encoding %d documents...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via normalized inner-product
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with METADATA_PATH.open("wb") as f:
        pickle.dump(catalog, f)

    logger.info(
        "Wrote index (%d vectors, dim=%d) to %s and metadata to %s",
        index.ntotal,
        dim,
        FAISS_INDEX_PATH,
        METADATA_PATH,
    )


if __name__ == "__main__":
    build()
