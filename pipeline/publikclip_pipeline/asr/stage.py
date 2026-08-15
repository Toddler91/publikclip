"""ASR + forced alignment via whisperX (BSD-2-Clause, pinned 3.8.6).

Word-level timestamps are the substrate for everything downstream: captions,
[laughs] tag placement, prosodic emphasis, long-pause detection, ducking,
and sentence-snapped candidate boundaries.

Model choice: large-v3-turbo int8 by default — near-parity accuracy with
large-v3 at a fraction of the compute, which is what makes local-first
viable on Apple Silicon. Silero VAD (MIT) instead of whisperX's bundled
pyannote VAD checkpoint, whose license the research flagged as unresolved.

The stage records wall-clock + realtime factor into its checkpoint — the M1
gate's Apple Silicon benchmark comes from real runs, not synthetic tests.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

from .. import config
from ..jobs.queue import Stage, StageContext, StageError
from . import chunks

ASR_MODEL = "large-v3-turbo"
COMPUTE_TYPE = "int8"
BATCH_SIZE = 8


def _point_caches_at_home() -> None:
    """All model caches live under PUBLIKCLIP_HOME so 'delete the app data
    dir' is a complete uninstall."""
    hf_home = config.models_dir() / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TORCH_HOME", str(config.models_dir() / "torch"))


class AsrStage(Stage):
    name = "asr"
    schema_version = 1

    def run(self, ctx: StageContext) -> dict:
        ingest = ctx.prior.get("ingest") if ctx.prior else None
        if not ingest:
            raise StageError("ASR needs the ingest stage output.")
        audio_path = Path(ingest["audio_path"])
        if not audio_path.exists():
            raise StageError("Analysis audio missing — re-run ingest.")

        _point_caches_at_home()
        ctx.emit(-1, "Loading speech model (downloads ~1.6 GB on first run)…")
        import torch  # deferred: heavy import
        import whisperx

        device = "cpu"  # ctranslate2 has no MPS backend; int8 CPU is the local path
        audio = whisperx.load_audio(str(audio_path))
        duration = float(len(audio)) / config.AUDIO_SR

        # Resumable unit of work. Long streams are the normal case here, and a
        # single uninterruptible hour of CPU is what made every kill expensive.
        spans = chunks.plan(audio, config.AUDIO_SR)
        partial = chunks.Partial.load_or_new(
            ctx.job_dir / chunks.PARTIAL_NAME,
            chunks.fingerprint(audio_path, len(audio)),
            spans,
        )
        partial_resumed = partial.transcribe_done() > 0 or partial.align_done() > 0
        if partial_resumed:
            ctx.emit(-1, f"Resuming — {partial.transcribe_done()}/{len(spans)} chunks already done")

        def samples(index: int):
            start, end = spans[index]
            return audio[int(start * config.AUDIO_SR):int(end * config.AUDIO_SR)]

        # -- pass 1: transcribe ------------------------------------------
        transcribe_secs = 0.0
        if partial.transcribe_done() < len(spans):
            t0 = time.monotonic()
            model = whisperx.load_model(
                ASR_MODEL, device, compute_type=COMPUTE_TYPE, vad_method="silero"
            )
            for index in range(len(spans)):
                if index in partial.transcribed:
                    continue
                ctx.emit(index / len(spans), f"Transcribing {index + 1}/{len(spans)}…")
                result = model.transcribe(
                    samples(index), batch_size=BATCH_SIZE, language=partial.language
                )
                # Detected once, then pinned: per-chunk detection can drift
                # mid-stream and silently switch languages on a quiet chunk.
                partial.language = partial.language or result.get("language", "en")
                partial.transcribed[index] = result.get("segments", [])
                partial.save()
            del model
            gc.collect()
            transcribe_secs = time.monotonic() - t0

        language = partial.language or "en"

        # -- pass 2: align -----------------------------------------------
        align_secs = 0.0
        if partial.align_done() < len(spans):
            t1 = time.monotonic()
            align_model, align_meta = whisperx.load_align_model(
                language_code=language, device=device
            )
            for index in range(len(spans)):
                if index in partial.aligned:
                    continue
                ctx.emit(index / len(spans), f"Aligning words {index + 1}/{len(spans)}…")
                segs = partial.transcribed.get(index) or []
                if not segs:
                    partial.aligned[index] = []
                else:
                    out = whisperx.align(
                        segs, align_model, align_meta, samples(index), device,
                        return_char_alignments=False,
                    )
                    partial.aligned[index] = out["segments"]
                partial.save()
            del align_model
            gc.collect()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            align_secs = time.monotonic() - t1

        segments = partial.merged_segments()
        word_count = sum(len(s["words"]) for s in segments)
        if word_count == 0:
            raise StageError(
                "No speech was found in this video. publikclip needs dialogue to find moments."
            )

        # The stage checkpoint now carries everything the partial held; keeping
        # both would leave a stale half-transcript to trip over on re-runs.
        partial.discard()

        total = transcribe_secs + align_secs
        return {
            "language": language,
            "model": ASR_MODEL,
            "compute_type": COMPUTE_TYPE,
            "segments": segments,
            "word_count": word_count,
            "chunks": len(spans),
            "benchmark": {
                "audio_sec": round(duration, 1),
                # Time spent in *this* run: a resumed job did part of the work
                # in an earlier process, so these are not comparable across
                # resumes and realtime_factor is only meaningful for one pass.
                "transcribe_sec": round(transcribe_secs, 1),
                "align_sec": round(align_secs, 1),
                "resumed": partial_resumed,
                "realtime_factor": round(duration / total, 2) if total > 0 else None,
            },
        }
