"""FAISS-backed semantic retriever with URL/name validation."""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, METADATA_PATH

logger = logging.getLogger(__name__)


class Retriever:
    """Loads the prebuilt FAISS index once and serves nearest-neighbor queries."""

    def __init__(self) -> None:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Index not built. Run: python -m scripts.build_index "
                f"(expected {FAISS_INDEX_PATH} and {METADATA_PATH})"
            )
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Loading FAISS index: %s", FAISS_INDEX_PATH)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with METADATA_PATH.open("rb") as f:
            self.catalog: list[dict] = pickle.load(f)

        self._url_index: dict[str, dict] = {
            item["link"]: item for item in self.catalog if item.get("link")
        }
        self._name_index: dict[str, dict] = {
            item["name"].lower(): item for item in self.catalog if item.get("name")
        }
        logger.info("Retriever ready: %d assessments indexed", len(self.catalog))

    def encode(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        return vec

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not query.strip():
            return []
        vec = self.encode(query)
        _scores, indices = self.index.search(vec, top_k)
        results: list[dict] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.catalog):
                results.append(self.catalog[int(idx)])
        return results

    def is_valid_url(self, url: str) -> bool:
        return url in self._url_index

    def get_by_url(self, url: str) -> dict | None:
        return self._url_index.get(url)

    def get_by_name(self, name: str) -> dict | None:
        if not name:
            return None
        lower = name.lower().strip()
        if lower in self._name_index:
            return self._name_index[lower]
        # fuzzy: substring match (catalog names tend to be unique enough)
        for catalog_name, item in self._name_index.items():
            if lower in catalog_name or catalog_name in lower:
                return item
        return None


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Singleton accessor — loads the index exactly once per process."""
    return Retriever()
