from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


def save_generation_cache(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON-serializable generation payload to a versioned cache file."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump({"version": CACHE_VERSION, "payload": payload}, handle, ensure_ascii=False, indent=2)


def load_generation_cache(path: str | Path) -> dict[str, Any] | None:
    """Load a generation payload, returning None when the cache file is absent."""
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    with cache_path.open(encoding="utf-8") as handle:
        cached = json.load(handle)
    if not isinstance(cached, dict) or cached.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported generation cache format in {cache_path}.")
    return cached["payload"]
