"""Per-candidate scoring progress, persisted as it happens.

Scoring calls the LLM once per candidate — 35+ on an hour-long stream — and a
quota that runs out at candidate 19 used to discard all nineteen and fail the
stage. They are the expensive part of the run, so they are written to disk as
they complete:

    <job_dir>/score.partial.json

The file is keyed on the candidate set it was built against, so re-running the
candidates stage invalidates it rather than pairing scores with the wrong
windows. Entries are stored by candidate index.

The LLM's own response cache already makes an identical re-call free, but that
cache is a convenience that can be cleared; this is the record that decides
whether a run can continue where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PARTIAL_NAME = "score.partial.json"
PARTIAL_VERSION = 1


def candidates_fingerprint(candidates: list[dict]) -> str:
    """Identity of the candidate set: count plus each window's bounds."""
    h = hashlib.sha256()
    h.update(str(len(candidates)).encode())
    for cand in candidates:
        h.update(f"{cand.get('start', 0):.3f}-{cand.get('end', 0):.3f};".encode())
    return h.hexdigest()[:16]


@dataclass
class PartialScores:
    path: Path
    fingerprint: str
    entries: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def load_or_new(cls, path: Path, fingerprint: str) -> "PartialScores":
        fresh = cls(path=path, fingerprint=fingerprint)
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return fresh
        if raw.get("version") != PARTIAL_VERSION or raw.get("fingerprint") != fingerprint:
            return fresh  # different candidates; these scores are not theirs
        entries = raw.get("entries") or {}
        if isinstance(entries, dict):
            fresh.entries = {int(k): v for k, v in entries.items()}
        return fresh

    def save(self) -> None:
        payload = {
            "version": PARTIAL_VERSION,
            "fingerprint": self.fingerprint,
            "updated_at": time.time(),
            "entries": {str(k): v for k, v in self.entries.items()},
        }
        tmp = self.path.with_suffix(".partial.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self.path)

    def discard(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def __len__(self) -> int:
        return len(self.entries)

    def ordered(self) -> list[dict]:
        """Scored entries in candidate order."""
        return [self.entries[i] for i in sorted(self.entries)]
