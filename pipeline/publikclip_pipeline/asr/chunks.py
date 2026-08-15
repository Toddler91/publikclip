"""Chunk boundaries and the partial-transcript store for resumable ASR.

Transcribing a 2.6 h stream is an hour of CPU, and until now it checkpointed
only at the very end: any interruption threw all of it away. This splits the
audio into windows that are checkpointed as they complete, so a kill costs one
chunk rather than the whole run.

Boundaries are not placed at fixed offsets. A cut through the middle of a word
loses it from both sides — the tail is truncated in one chunk and the head is
missing context in the next — so each target offset is nudged to the quietest
short frame within a search window around it. Silence is where a cut is free.

The partial store is a sibling of the stage checkpoint:

    <job_dir>/asr.partial.json

It is keyed by an audio fingerprint; if the analysis audio is regenerated the
partial is discarded rather than stitched onto the wrong media. Segment times
inside it stay *chunk-relative*, because that is what whisperX's aligner needs
when it is handed a chunk's samples; they are offset to absolute time only at
the merge step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PARTIAL_NAME = "asr.partial.json"
PARTIAL_VERSION = 1

# ~10 min of audio per chunk: long enough that per-chunk model overhead stays
# negligible, short enough that losing one to a kill is a minor setback.
CHUNK_SEC = 600.0
# How far from the target offset we will move to find silence.
SEARCH_SEC = 15.0
# Frame length whose energy we minimise. Shorter than the shortest natural
# pause between words, so a real gap can be located inside it.
FRAME_SEC = 0.4
# Never emit a chunk shorter than this (guards the tail).
MIN_CHUNK_SEC = 60.0


def _rms_frames(samples: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS per non-overlapping frame; trailing partial frame dropped."""
    usable = (len(samples) // frame_len) * frame_len
    if usable == 0:
        return np.array([])
    frames = samples[:usable].reshape(-1, frame_len).astype(np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _quietest_time(audio: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Centre of the quietest FRAME_SEC frame in [lo, hi]."""
    frame_len = max(1, int(FRAME_SEC * sr))
    start = max(0, int(lo * sr))
    stop = min(len(audio), int(hi * sr))
    if stop - start < frame_len:
        return (lo + hi) / 2
    rms = _rms_frames(audio[start:stop], frame_len)
    if rms.size == 0:
        return (lo + hi) / 2
    idx = int(np.argmin(rms))
    return (start + idx * frame_len + frame_len / 2) / sr


def plan(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Chunk spans as (start, end) seconds, snapped to quiet points."""
    total = len(audio) / float(sr)
    if total <= CHUNK_SEC + MIN_CHUNK_SEC:
        return [(0.0, total)]

    bounds = [0.0]
    target = CHUNK_SEC
    while total - target > MIN_CHUNK_SEC:
        lo = max(bounds[-1] + MIN_CHUNK_SEC, target - SEARCH_SEC)
        hi = min(total - MIN_CHUNK_SEC, target + SEARCH_SEC)
        cut = _quietest_time(audio, sr, lo, hi) if hi > lo else target
        cut = min(max(cut, lo), hi)
        bounds.append(cut)
        target = cut + CHUNK_SEC
    bounds.append(total)
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0.01]


def fingerprint(audio_path: Path, sample_count: int) -> str:
    """Cheap identity for the analysis audio: size + sample count. Enough to
    notice 're-ran ingest on different media' without hashing 150 M samples."""
    try:
        size = audio_path.stat().st_size
    except OSError:
        size = -1
    return f"{size}:{sample_count}"


@dataclass
class Partial:
    """Per-chunk transcripts and alignments, persisted between runs."""

    path: Path
    fingerprint: str
    spans: list[tuple[float, float]]
    language: str | None = None
    transcribed: dict[int, list[dict]] = field(default_factory=dict)
    aligned: dict[int, list[dict]] = field(default_factory=dict)

    # -- persistence ------------------------------------------------------
    @classmethod
    def load_or_new(cls, path: Path, fingerprint: str, spans: list[tuple[float, float]]) -> "Partial":
        """Reuse a partial only if it matches this audio and chunk plan."""
        fresh = cls(path=path, fingerprint=fingerprint, spans=spans)
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return fresh
        if raw.get("version") != PARTIAL_VERSION or raw.get("fingerprint") != fingerprint:
            return fresh
        saved_spans = [tuple(s) for s in raw.get("spans", [])]
        if len(saved_spans) != len(spans) or any(
            abs(a[0] - b[0]) > 0.01 or abs(a[1] - b[1]) > 0.01
            for a, b in zip(saved_spans, spans)
        ):
            return fresh  # plan changed; stale work cannot be trusted
        fresh.language = raw.get("language")
        fresh.transcribed = {int(k): v for k, v in (raw.get("transcribed") or {}).items()}
        fresh.aligned = {int(k): v for k, v in (raw.get("aligned") or {}).items()}
        return fresh

    def save(self) -> None:
        payload = {
            "version": PARTIAL_VERSION,
            "fingerprint": self.fingerprint,
            "updated_at": time.time(),
            "spans": [list(s) for s in self.spans],
            "language": self.language,
            "transcribed": {str(k): v for k, v in self.transcribed.items()},
            "aligned": {str(k): v for k, v in self.aligned.items()},
        }
        tmp = self.path.with_suffix(".partial.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self.path)

    def discard(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    # -- progress ---------------------------------------------------------
    @property
    def n_chunks(self) -> int:
        return len(self.spans)

    def transcribe_done(self) -> int:
        return sum(1 for i in range(self.n_chunks) if i in self.transcribed)

    def align_done(self) -> int:
        return sum(1 for i in range(self.n_chunks) if i in self.aligned)

    # -- merge ------------------------------------------------------------
    def merged_segments(self) -> list[dict]:
        """Aligned chunks stitched back into one absolute-time transcript."""
        out: list[dict] = []
        for index, (start, _end) in enumerate(self.spans):
            for seg in self.aligned.get(index, []):
                words = [
                    {
                        "word": w.get("word", "").strip(),
                        "start": round(float(w["start"]) + start, 3),
                        "end": round(float(w["end"]) + start, 3),
                        "score": round(float(w.get("score", 0.0)), 3),
                    }
                    for w in seg.get("words", [])
                    if w.get("start") is not None and w.get("end") is not None
                ]
                if seg.get("start") is None or seg.get("end") is None:
                    continue
                out.append({
                    "start": round(float(seg["start"]) + start, 3),
                    "end": round(float(seg["end"]) + start, 3),
                    "text": (seg.get("text") or "").strip(),
                    "words": words,
                })
        out.sort(key=lambda s: s["start"])
        return out
