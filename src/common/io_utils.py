"""Small shared I/O helpers (JSONL, hashing, config loading)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Iterable, Iterator

# make `src` importable when scripts are run directly (python src/stage4_survey/run_survey.py ...)
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict], mode: str = "w") -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class JsonlAppender:
    """Thread-safe append-only JSONL writer with flush per record (crash-safe resume)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self):
        with self._lock:
            self._f.close()


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj, indent: int = 2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int_hash(*parts, bits: int = 32) -> int:
    """Deterministic integer hash of string parts (for per-agent/condition RNG seeds)."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h, 16) % (1 << bits)
