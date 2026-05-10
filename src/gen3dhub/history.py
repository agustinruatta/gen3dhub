"""Append-only JSONL log of past runs.

Stored at `<cache_root>/history.jsonl` (one JSON object per line). The format
is line-oriented so a partially-written file (e.g. crash mid-write) doesn't
corrupt earlier entries — readers skip malformed lines.

Why not SQLite: this log is small (one entry per run, dozens to thousands of
lines over a project's lifetime) and append-mostly. JSONL is human-greppable
and survives "I deleted the binary" — the user can still read their history
with `cat ~/.cache/gen3dhub/history.jsonl | jq`.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gen3dhub.config import Paths


@dataclass
class HistoryEntry:
    """One row in the history log. All fields are JSON-serializable."""

    id: str            # short unique id, e.g. "20260510-150234-abc123"
    timestamp: str     # ISO-8601 UTC ("Z" suffix)
    model: str
    inputs: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    output: str | None = None    # produced GLB path, if any
    preview: str | None = None   # preview PNG path, if any
    duration_s: float = 0.0      # wall time of the whole run including preview
    exit_code: int = 0           # 0 = success; matches the CLI's exit contract


def _log_path(paths: Paths) -> Path:
    return paths.cache_root / "history.jsonl"


def make_id() -> str:
    """Stable-sort-friendly id: timestamp prefix + 6 random hex chars."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(paths: Paths, entry: HistoryEntry) -> None:
    """Atomic-ish append. We open for append so concurrent runs don't clobber
    each other (POSIX guarantees writes < PIPE_BUF are atomic, and our entries
    are well under that)."""
    path = _log_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(entry), default=str) + "\n"
    with path.open("a") as f:
        f.write(line)


def read_all(paths: Paths) -> list[HistoryEntry]:
    """Return all entries in chronological order. Malformed lines are skipped
    silently — a corrupted entry shouldn't keep the user from seeing the rest
    of their history."""
    path = _log_path(paths)
    if not path.exists():
        return []
    entries: list[HistoryEntry] = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                entries.append(HistoryEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def find(paths: Paths, prefix: str) -> HistoryEntry | None:
    """Find the most recent entry whose id starts with `prefix`. Allows users
    to type just a few chars instead of the full id."""
    for entry in reversed(read_all(paths)):
        if entry.id.startswith(prefix):
            return entry
    return None
