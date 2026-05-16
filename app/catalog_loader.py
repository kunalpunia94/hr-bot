"""Download and cache the SHL product catalog JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from app.config import CATALOG_PATH, CATALOG_URL, DATA_DIR

logger = logging.getLogger(__name__)


def download_catalog(url: str = CATALOG_URL, save_path: Path = CATALOG_PATH) -> list[dict]:
    """Fetch catalog from the SHL endpoint and write it to disk.

    The upstream JSON occasionally contains stray control characters inside
    description strings, so we parse with strict=False.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading catalog from %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = json.loads(resp.text, strict=False)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved %d assessments to %s", len(data), save_path)
    return data


def load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        return download_catalog(save_path=path)
    return json.loads(path.read_text(), strict=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items = download_catalog()
    print(f"OK — {len(items)} assessments cached at {CATALOG_PATH}")
