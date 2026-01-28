from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


CACHE_DIR = Path(".cache/powercurves")


def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def make_key(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def read_cache(key: str) -> Optional[Dict[str, Any]]:
    path = ensure_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_cache(key: str, data: Dict[str, Any]) -> None:
    path = ensure_cache_dir() / f"{key}.json"
    path.write_text(json.dumps(data, indent=2))
